import yaml
from io import StringIO
from conceptLogic import StandardLogic, isInstanceOf, StringConcept, ConstructedAbstraction, ReferencedAbstraction, CodedConceptClass, newIdentityConcept, NumberConcept, TripleTrueAssertion, writeTriples, readTriples, getConceptName
from conceptLogic import writeDistinctConnection, StringConcept, readDistinctConnection, IdentityConcept, semanticConnectionsNotValid, hasConceptClass
from pathlib import Path

directory_path = Path(__file__).parent
root_path = (directory_path / ".." / "..").resolve()
referenced_R30_value_path = root_path / "master_thesis" / "semantic_paper" / "data" / "collect_R30_values" / "referenced_papers" / "parameters.yaml"

MLMPrefix = b"masterThesis.MLM."

# Create constructed concept definitions

hasArxivId = newIdentityConcept("hasArxivId", MLMPrefix)
class ArxivPaper(metaclass=CodedConceptClass):
    """
    A concept that represents a paper on arxiv.org
    It has a string as conceptContent that represents the arxiv id of the paper.
    """
    prefix = MLMPrefix
    def getConnectionsFromContent(content, conceptLogic):
        return writeDistinctConnection(StringConcept(content, conceptLogic), hasArxivId, conceptLogic)
    def getContentFromConnections(semanticConnections, conceptLogic):
        string = readDistinctConnection(hasArxivId, semanticConnections, conceptLogic)
        if not hasConceptClass(string, StringConcept):
            raise semanticConnectionsNotValid()
        return string.content
    def contentValid(content, conceptLogic):
        return type(content) == str

hasAutorityId = newIdentityConcept("hasAuthorityId", MLMPrefix)
class Autority(metaclass=CodedConceptClass):
    """
    A concept that represents an authority that can be used to reference a paper.
    It has a bytes object as conceptContent that represents the id of the authority.
    """
    prefix = MLMPrefix
    def getConnectionsFromContent(content, conceptLogic):
        return writeDistinctConnection(IdentityConcept(content, conceptLogic), hasAutorityId, conceptLogic)
    def getContentFromConnections(semanticConnections, conceptLogic):
        id = readDistinctConnection(hasAutorityId, semanticConnections, conceptLogic)
        if not hasConceptClass(id, IdentityConcept):
            raise semanticConnectionsNotValid()
        return id.content
    def contentValid(content, conceptLogic):
        return type(content) == bytes

authorityOfSourceBasedClaim = newIdentityConcept("authorityOfSourceBasedClaim", MLMPrefix)
sourceOfSourceBasedClaim = newIdentityConcept("sourceOfSourceBasedClaim", MLMPrefix)
claimesSourceBased = newIdentityConcept("claimesSourceBased", MLMPrefix)
class SourceBasedClaim(metaclass=CodedConceptClass):
    """
    A concept that represents a claim that is based on a source.
    It has a tuple containing an authority, a source and a assertion as conceptContent.
    """
    prefix = MLMPrefix
    def getContentFromConnections(semanticConnections, conceptLogic):
        authority = readDistinctConnection(authorityOfSourceBasedClaim, semanticConnections, conceptLogic)
        if not hasConceptClass(authority, Autority):
            raise semanticConnectionsNotValid()
        source = readDistinctConnection(sourceOfSourceBasedClaim, semanticConnections, conceptLogic)
        assertion = readDistinctConnection(claimesSourceBased, semanticConnections, conceptLogic)
        #if not hasConceptClass(assertion, TripleTrueAssertion):
        #    raise semanticConnectionsNotValid()
        return (assertion, authority, source)
    def getConnectionsFromContent(content, conceptLogic):
        return frozenset([
            (None, claimesSourceBased.getConcept(conceptLogic), content[0]),
            (None, authorityOfSourceBasedClaim.getConcept(conceptLogic), content[1]),
            (None, sourceOfSourceBasedClaim.getConcept(conceptLogic), content[2])
        ])
    def contentValid(content, conceptLogic):
        return isinstance(content, tuple) and len(content) == 3 and hasConceptClass(content[1], Autority)  #and hasConceptClass(content[0], TripleTrueAssertion)

# Create StandardLogic
sl = StandardLogic()

#arxp = ArxivPaper.getConcept(sl)


# Create referenced abstractions

hasName = ReferencedAbstraction(MLMPrefix + b"hasName", sl)

trainedMLMParametersFormat = ReferencedAbstraction(MLMPrefix + b"trainedMLMParametersFormat", sl)
hasMLMParameterCount = ReferencedAbstraction(MLMPrefix + b"hasMLMParameterCount", sl)

trainedMLMParameters = ReferencedAbstraction(MLMPrefix + b"trainedMLMParemeters", sl)
hasTrainedMLMParametersFormat = ReferencedAbstraction(MLMPrefix + b"hasTrainedMLMParametersFormat", sl)

trainedSignalValueClassificationMLM = ReferencedAbstraction(MLMPrefix + b"trainedSignalValueClassificationMLM", sl)
includesTrainedMLMData = ReferencedAbstraction(MLMPrefix + b"includesTrainedMLMData", sl)

physicsBasedTopQuarkClassification = ReferencedAbstraction(MLMPrefix + b"physicsBasedTopQuarkClassification", sl)

signalValueClassificationApproximation = ReferencedAbstraction(MLMPrefix + b"signalValueClassificationApproximation", sl)
functionUsedForSignalValueClassification = ReferencedAbstraction(MLMPrefix + b"functionUsedForSignalValueClassification", sl)
targetSignalValueClassificationFunction = ReferencedAbstraction(MLMPrefix + b"targetSignalValueClassificationFunction", sl)

r30Evaluation = ReferencedAbstraction(MLMPrefix + b"r30Evaluation", sl)
r30EvaluationOf = ReferencedAbstraction(MLMPrefix + b"r30EvaluationOf", sl)

hasR30Value = ReferencedAbstraction(MLMPrefix + b"hasR30Value", sl)
# Iterate over data
with open(referenced_R30_value_path, "r") as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
"""
Format of the data:
dict(
    model=string : dict(
        "r30" : dict(
            value=int : list(source=string)
        ),
        "param" : dict(
            value=int : list(source=string)
        ),
)
"""
# Get all source strings
source_strings = set([source for model, parameters in data.items() for parameter, values in parameters.items() for value, sources in values.items() for source in sources])
"""
Format of the source strings:
"Paper: arxif.org/abs/" + arxiv_id + " Table: " + table_number + " Row " + row_number
"""
def get_arxiv_id(source_string):
    return source_string.split("arxiv.org/abs/")[1].split(" Table:")[0]
arxiv_ids = set([get_arxiv_id(source_string) for source_string in source_strings])
papers = []

for arxiv_id in arxiv_ids:
    papers.append(ArxivPaper(arxiv_id, sl))


authority = Autority(MLMPrefix + b"R30ExtractionAuthority", sl)

claimes = []

for modelName, parameters in data.items():
    byteModelName = modelName.encode("utf-8")
    modelParametersFormat = ReferencedAbstraction((MLMPrefix + b"models." + byteModelName + b".parametersFormat", [(None, isInstanceOf, trainedMLMParametersFormat)]), sl)
    modelParameters = ReferencedAbstraction((MLMPrefix + b"models." + byteModelName + b".parameters", [(None, isInstanceOf, trainedMLMParameters), (None, hasTrainedMLMParametersFormat, modelParametersFormat)]), sl)
    model = ReferencedAbstraction((MLMPrefix + b"models." + byteModelName, [(None, isInstanceOf, trainedSignalValueClassificationMLM), (None, includesTrainedMLMData, modelParameters)]), sl)
    modelApproximation = ReferencedAbstraction((MLMPrefix + b"models." + byteModelName + b".approximation", [(None, isInstanceOf, signalValueClassificationApproximation), (None, functionUsedForSignalValueClassification, model), (None, targetSignalValueClassificationFunction, physicsBasedTopQuarkClassification)]), sl)
    modelR30Evaluation = ReferencedAbstraction((MLMPrefix + b"models." + byteModelName + b".r30Evaluation", [(None, isInstanceOf, r30Evaluation), (None, r30EvaluationOf, modelApproximation)]), sl)
    nameStringConcept = StringConcept(modelName, sl)
    hasNameAssertion = TripleTrueAssertion((model, hasName, nameStringConcept), sl)
    for source in [source for r30, sources in parameters.get("r30", {}).items() for source in sources] + [source for param, sources in parameters.get("param", {}).items() for source in sources]:
        arxiv_id = get_arxiv_id(source)
        arxivPaper = ArxivPaper(arxiv_id, sl)
        sourceBasedClaim = SourceBasedClaim((hasNameAssertion, authority, arxivPaper), sl)
        claimes.append(sourceBasedClaim)
    if "r30" in parameters:
        for r30, sources in parameters["r30"].items():
            r30 = int(r30)
            r30 = NumberConcept(r30, sl)
            r30TripleTrueAssertion = TripleTrueAssertion((modelR30Evaluation, hasR30Value, r30), sl)
            for source in sources:
                arxiv_id = get_arxiv_id(source)
                arxivPaper = ArxivPaper(arxiv_id, sl)
                r30Claim = SourceBasedClaim((r30TripleTrueAssertion, authority, arxivPaper), sl)
                claimes.append(r30Claim)
    if "param" in parameters:
        for param, sources in parameters["param"].items():
            param = int(param)
            param = NumberConcept(param, sl)
            paramTripleTrueAssertion = TripleTrueAssertion((modelParametersFormat, hasMLMParameterCount, param), sl)
            for source in sources:
                arxiv_id = get_arxiv_id(source)
                arxivPaper = ArxivPaper(arxiv_id, sl)
                paramClaim = SourceBasedClaim((paramTripleTrueAssertion, authority, arxivPaper), sl)
                claimes.append(paramClaim)
""""""
# Write the concepts to a file and read them back in

concepts = sl.getLoadedConcepts()
with open(directory_path / "R30.ttl", "w") as s:
        writeTriples(concepts, s)
with open(directory_path / "R30.ttl", "r") as s:
        rt1 = readTriples(s, sl)
with open(directory_path / "R30.ttl", "r") as s:
        rt2 = readTriples(s, sl)
        print(all([c in rt1.values() for c in concepts]))
        for c in concepts.difference(set(rt1.values())):
            print(getConceptName(c))
        print(set(rt1) == set(rt2))
